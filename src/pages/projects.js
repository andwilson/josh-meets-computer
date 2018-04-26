import React from "react";
import Link from "gatsby-link";
import Helmet from "react-helmet";
import styled from "styled-components";
import Img from "gatsby-image";

import CategoryHeader from "../components/CategoryHeader";

const SLink = styled(Link)`
  transition: all 0.2s;
  &:hover {
  }
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  @media (min-width: 600px) {
    grid-template-columns: 1fr 1fr;
  }
  grid-gap: 1em;
  margin-top: 1em;
`;

const Project = styled.div`
  border-radius: 10px;
  position: relative;
  transition: all 0.5s ease;
  &:hover {
    opacity: 0.9;
    -webkit-box-shadow: 0px 0px 2px grey;
    -moz-box-shadow: 0px 0px 2px grey;
    box-shadow: 0px 0px 2px grey;
  }
`;

const SImg = styled(Img)`
  width: "100%";
  height: "auto";
  z-index: -1;
  border: 1px solid grey;
  border-radius: 10px;
`;

const Description = styled.div`
  position: absolute;
  padding: 1em;
  @media (min-width: 800px) {
    padding: 2em;
  }
  z-index: 2;
  top: 0;
  left: 0;
  > h2 {
    color: white;
    margin: 0;
  }
  > p {
    color: white;
    font-size: 12px;
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
                  <SLink to={post.node.frontmatter.path}>
                    <SImg
                      sizes={
                        post.node.frontmatter.thumbnail.childImageSharp.sizes
                      }
                    />
                    <Description>
                      <h2>{post.node.frontmatter.title}</h2>
                      <p>{post.node.frontmatter.date}</p>
                      <p
                        dangerouslySetInnerHTML={{ __html: post.node.excerpt }}
                      />
                    </Description>
                  </SLink>
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
                  maxWidth: 600
                  maxHeight: 400
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
