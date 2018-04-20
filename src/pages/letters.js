import React from "react";
import Link from "gatsby-link";
import Helmet from "react-helmet";
import styled from "styled-components";

import CategoryHeader from "../components/CategoryHeader";

const Post = styled.div`
  > h3 {
    margin-bottom: 5px;
  }
  > small {
    font-family: roboto;
    font-size: 14px;
    color: grey;
    font-style: italic;
  }
  > p {
    margin-top: 10px;
  }
`;

const SLink = styled(Link)`
  text-decoration: none;
  color: #28aa55;
  &:hover {
    color: #23984c;
  }
`;

class Letters extends React.Component {
  render() {
    const posts = this.props.data.allMarkdownRemark.edges;
    const categoryTitle = "Letters";
    const categoryDescription = "Below are the weekly letters I've written so far. A little while ago I started writing weekly letters to my loved ones as a means to keep everyone updated and informed of my whereabouts and happenings. Soon I discovered that the process of writing about my week was beneficial in so many ways! It was meditative. It allowed me time to process what I've experienced. It helped me appreciate the good things that happened in a regular week. It gave me a permanent, detailed repository of the story of my life.";

    return (
      <div>
        <Helmet title={categoryTitle} />
        <CategoryHeader
          title={categoryTitle}
          description={categoryDescription}
          data= {this.props.data}
        />
        {posts.map(post => {
          if (
            post.node.path !== "/404/"
          ) {
            return (
              <Post key={post.node.frontmatter.path}>
                <h3>
                  <SLink to={post.node.frontmatter.path}>
                    {post.node.frontmatter.title}
                  </SLink>
                </h3>
                <small>{post.node.frontmatter.date}</small>
                <p dangerouslySetInnerHTML={{ __html: post.node.excerpt }} />
              </Post>
            );
          }
        })}
      </div>
    );
  }
}

export default Letters;

export const pageQuery = graphql`
  query LettersQuery {
    allMarkdownRemark(
      sort: { fields: [frontmatter___date], order: DESC },
      filter: {frontmatter: {category: {eq: "Letters"}}}
    ) {
      totalCount
      edges {
        node {
          excerpt(pruneLength: 250)
          frontmatter {
            path
            date(formatString: "MMMM D, YYYY")
            title
            category
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
