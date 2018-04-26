import React from "react";
import Link from "gatsby-link";
import Helmet from "react-helmet";
import styled from "styled-components";
import Img from "gatsby-image";

import CategoryHeader from "../components/CategoryHeader";

const SLink = styled(Link)`
  text-decoration: none;
  color: white;
  transition: all 0.2s;
  &:hover {
    color: #e4e4e4;
  }
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: 50% 50%;
  grid-gap: 1em;
`;

const SImg = styled(Img)``;

const Project = styled.div`
  position: relative;
`;

const Description = styled.div`
  position: absolute;
  padding: 1em;
  z-index: 2;
  top: 0;
  left: 0;
  > h3 {
    margin: 0;
  }
  > p {
    color: white;
  }
  > p:nth-child(2) {
    font-style: italic;
    color: #e4e4e4;
  }
`;

class Projects extends React.Component {
  render() {
    const posts = this.props.data.allMarkdownRemark.edges;
    const categoryTitle = "Projects";
    const categoryDescription =
      "Below highlight a collection of projects I've worked on. A few years ago I decided to make Artificial Intelligence my single focus as an engineer, and recently refined my focused on applying AI/ML algorithms and systems to relationships and experiences. A bit of a catch-all for my ventures in this space!";

    return (
      <div>
        <Helmet title={categoryTitle} />
        <CategoryHeader
          title={categoryTitle}
          description={categoryDescription}
          data={this.props.data}
        />
        <Grid>
          {posts.map(post => {
            if (post.node.path !== "/404/") {
              return (
                <Project key={post.node.frontmatter.path}>
                  <SImg
                    sizes={
                      post.node.frontmatter.thumbnail.childImageSharp.sizes
                    }
                    style={{
                      width: "100%",
                      height: "auto",
                      zIndex: -1
                    }}
                  />
                  <Description>
                    <h3>
                      <SLink to={post.node.frontmatter.path}>
                        {post.node.frontmatter.title}
                      </SLink>
                    </h3>
                    <p>{post.node.frontmatter.date}</p>
                    <p
                      dangerouslySetInnerHTML={{ __html: post.node.excerpt }}
                    />
                  </Description>
                </Project>
              );
            }
          })}
        </Grid>
      </div>
    );
  }
}

export default Projects;

export const pageQuery = graphql`
  query ProjectsQuery {
    allMarkdownRemark(
      sort: { fields: [frontmatter___date], order: DESC }
      filter: { frontmatter: { category: { eq: "Projects" } } }
    ) {
      totalCount
      edges {
        node {
          excerpt(pruneLength: 300)
          frontmatter {
            path
            date(formatString: "MMMM D, YYYY")
            title
            category
            thumbnail {
              childImageSharp {
                sizes(
                  duotone: { highlight: "#28aa55", shadow: "#1d7f3f" }
                  maxWidth: 800
                ) {
                  ...GatsbyImageSharpSizes
                }
              }
            }
          }
        }
      }
    }
    avatar: imageSharp(id: { regex: "/avatar.jpg/" }) {
      sizes(maxWidth: 500, grayscale: false) {
        ...GatsbyImageSharpSizes_tracedSVG
      }
    }
  }
`;
